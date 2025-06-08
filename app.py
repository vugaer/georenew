from flask import Flask, render_template, request, jsonify, send_file, abort
import os
import json
import cv2
import numpy as np
from io import BytesIO

app = Flask(__name__)

PROJECTS_JSON = 'data/projects.json'
PROJECTS_DIR = 'static/projects'

# Load Mapbox API key from environment variable or config
MAPBOX_API_KEY = "pk.eyJ1IjoiZmFyaWRkYWRhc2hvdjU1NiIsImEiOiJjbWJtaHd1N2gxOXk0MmpyMWZldDZoYjM1In0.8sGYsdNMpqKRDlzCUy-epg"

def load_projects():
    if not os.path.exists(PROJECTS_JSON):
        return {}
    with open(PROJECTS_JSON, 'r') as f:
        return json.load(f)

def save_projects(projects):
    with open(PROJECTS_JSON, 'w') as f:
        json.dump(projects, f, indent=2)

def ensure_project_dir(name):
    path = os.path.join(PROJECTS_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path

def generate_heatmap(image):
    # Convert to grayscale and apply color map for heatmap effect
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return heatmap

def generate_mask(image, threshold=128):
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # Convert mask to 3-channel for consistent display
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask_color

@app.route('/')
def index():
    return render_template('index.html', api_key=MAPBOX_API_KEY)

@app.route('/get_projects')
def get_projects():
    projects = load_projects()
    return jsonify(projects)

@app.route('/get_image', methods=['POST'])
def get_image():
    lat = float(request.form.get('lat'))
    lon = float(request.form.get('lon'))
    zoom = int(request.form.get('zoom'))
    name = request.form.get('name')
    width = int(request.form.get('width'))
    height = int(request.form.get('height'))
    threshold = int(request.form.get('threshold', 128))

    # Validate inputs
    if not name or width <= 0 or height <= 0:
        return 'Invalid parameters', 400

    project_path = ensure_project_dir(name)

    # Use Mapbox Static API to download satellite image
    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
        f"{lon},{lat},{zoom},0/{width}x{height}"
        f"?access_token={MAPBOX_API_KEY}"
    )

    import requests
    response = requests.get(url)
    if response.status_code != 200:
        return "Failed to download satellite image", 500

    # Save original image
    img_path = os.path.join(project_path, 'image.png')
    with open(img_path, 'wb') as f:
        f.write(response.content)

    # Read image with OpenCV
    img_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Generate and save heatmap
    heatmap = generate_heatmap(image)
    heatmap_path = os.path.join(project_path, 'heatmap.png')
    cv2.imwrite(heatmap_path, heatmap)

    # Generate and save mask with given threshold
    mask = generate_mask(image, threshold)
    mask_path = os.path.join(project_path, 'mask.png')
    cv2.imwrite(mask_path, mask)

    # Save project metadata (lat/lon)
    projects = load_projects()
    projects[name] = {'lat': lat, 'lon': lon}
    save_projects(projects)

    return 'OK'

@app.route('/load_project/<name>/<image_type>')
def load_project_image(name, image_type):
    # Optional threshold query param for mask image
    threshold = request.args.get('threshold', default=None, type=int)

    project_path = os.path.join(PROJECTS_DIR, name)
    if not os.path.isdir(project_path):
        abort(404)

    if image_type == 'image':
        img_path = os.path.join(project_path, 'image.png')
    elif image_type == 'heatmap':
        img_path = os.path.join(project_path, 'heatmap.png')
    elif image_type == 'mask':
        # If threshold param is provided, regenerate mask dynamically
        if threshold is not None:
            orig_img_path = os.path.join(project_path, 'image.png')
            if not os.path.isfile(orig_img_path):
                abort(404)
            image = cv2.imread(orig_img_path)
            mask = generate_mask(image, threshold)
            # Return mask as PNG bytes
            _, buffer = cv2.imencode('.png', mask)
            return send_file(BytesIO(buffer.tobytes()), mimetype='image/png')
        else:
            img_path = os.path.join(project_path, 'mask.png')
    else:
        abort(404)

    if not os.path.isfile(img_path):
        abort(404)

    return send_file(img_path, mimetype='image/png')

@app.route('/rename_project', methods=['POST'])
def rename_project():
    old_name = request.form.get('old')
    new_name = request.form.get('new')

    if not old_name or not new_name:
        return 'Missing parameters', 400

    projects = load_projects()
    if old_name not in projects:
        return 'Project not found', 404
    if new_name in projects:
        return 'New project name already exists', 400

    old_path = os.path.join(PROJECTS_DIR, old_name)
    new_path = os.path.join(PROJECTS_DIR, new_name)

    try:
        os.rename(old_path, new_path)
        projects[new_name] = projects.pop(old_name)
        save_projects(projects)
        return 'OK'
    except Exception as e:
        return f'Error renaming project: {str(e)}', 500

@app.route('/delete_project/<name>', methods=['POST'])
def delete_project(name):
    import shutil
    projects = load_projects()
    if name not in projects:
        return 'Project not found', 404

    project_path = os.path.join(PROJECTS_DIR, name)
    try:
        shutil.rmtree(project_path)
        projects.pop(name)
        save_projects(projects)
        return 'OK'
    except Exception as e:
        return f'Error deleting project: {str(e)}', 500

if __name__ == '__main__':
    app.run(debug=True)
