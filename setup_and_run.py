import sys
import os
import subprocess

def setup_project():
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.append(src_path)
    print("Project setup complete. PYTHONPATH updated.")

def install_dependencies():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
        print("Dependencies installed.")
    else:
        print("No requirements.txt file found.")

def run_tests():
    subprocess.call(['pytest', os.path.join(os.path.dirname(__file__), 'tests')])

if __name__ == '__main__':
    setup_project()
    install_dependencies()
    run_tests()
