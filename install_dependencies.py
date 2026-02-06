import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("Installing required dependencies...")

packages = [
    'pandas',
    'numpy',
    'matplotlib',
    'seaborn',
    'scikit-learn',
    'openpyxl',
    'streamlit'
]

for package in packages:
    print(f"Installing {package}...")
    try:
        install_package(package)
        print(f"✓ {package} installed successfully")
    except Exception as e:
        print(f"✗ Error installing {package}: {e}")

print("\nAll dependencies installed!")
print("\nNow run: python adhd_streamlit_app.py")