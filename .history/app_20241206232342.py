from flask import Flask, render_template, request, url_for
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19

app = Flask(__name__)

# Load the VGG19 model
model = vgg19(pretrained=True).features
for param in model.parameters():
    param.requires_grad = False
model.eval()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Preprocess the input image
def preprocess_image(image_path, size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0).to(device)


# Compute Gram matrix
def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    gram = torch.mm(features, features.t())
    return gram / (c * h * w)


# Extract features from specific layers
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'ConvLayer_1',
            '5': 'ConvLayer_2',
            '10': 'ConvLayer_3',
            '19': 'ConvLayer_4',
            '28': 'ConvLayer_5'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_and_style():
    print("HERE")
    # Retrieve style choice
    style_choice = request.form.get('options')
    if not style_choice:
        return "No style option selected.", 400
    print("HERE")
    # Retrieve and save content image
    content_file = request.files.get('content_image')
    if not content_file:
        return "No content image uploaded.", 400

    content_path = "static/content.jpg"
    content_file.save(content_path)

    # Map style choices to style images
    if style_choice == "option1":
        style_path = "static/myStyleOne.jpg"
    elif style_choice == "option2":
        style_path = "static/myStyleTwo.jpg"
    elif style_choice == "option3":
        style_path = "static/myStyleThree.jpg"
    else:
        return "Invalid style option selected.", 400

    # Preprocess images
    content_tensor = preprocess_image(content_path)
    style_tensor = preprocess_image(style_path)

    # Initialize target image
    target = content_tensor.clone().requires_grad_(True)

    # Extract features
    content_features = get_features(content_tensor, model)
    style_features = get_features(style_tensor, model)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Specify layers and weights
    layers4content = ['ConvLayer_1', 'ConvLayer_4']
    layers4style = ['ConvLayer_1', 'ConvLayer_2', 'ConvLayer_3', 'ConvLayer_4', 'ConvLayer_5']
    weights4style = [1, 0.5, 0.5, 0.2, 0.1]

    # Optimize target image
    optimizer = torch.optim.RMSprop([target], lr=0.005)
    for step in range(1500):
        target_features = get_features(target, model)

        # Content loss
        content_loss = sum(
            torch.mean((target_features[layer] - content_features[layer]) ** 2)
            for layer in layers4content
            if layer in target_features
        )

        # Style loss
        style_loss = sum(
            weights4style[layers4style.index(layer)] *
            torch.mean((gram_matrix(target_features[layer]) - style_grams[layer]) ** 2)
            for layer in layers4style
            if layer in target_features
        )

        # Total loss
        total_loss = 1e6 * style_loss + content_loss

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Save and return styled image
    styled_image_path = "static/styled_image.jpg"
    final_image = torch.sigmoid(target.clone().detach()).squeeze()
    final_image = final_image.cpu().clamp(0, 1)
    transforms.ToPILImage()(final_image).save(styled_image_path)

    return render_template('index.html', styled_image=url_for('static', filename='styled_image.jpg'))


if __name__ == '__main__':
    app.run(port=3000, debug=True)
