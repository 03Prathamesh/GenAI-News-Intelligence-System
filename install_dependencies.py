import subprocess
import sys
import os

def check_python_version():
    print("Checking Python version...")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("⚠️  Python 3.8 or higher is recommended")
        return True

def install_package(package, retry_count=2):
    for attempt in range(retry_count + 1):
        try:
            print(f"\nAttempt {attempt + 1}/{retry_count + 1}: Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Successfully installed {package}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package} (Attempt {attempt + 1})")
            if attempt < retry_count:
                print("Retrying with different version...")
                # Try without version specifier
                if "==" in package:
                    package_name = package.split("==")[0]
                    package = package_name
            else:
                print(f"⚠️  Could not install {package}")
                return False
    return False

def check_installation():
    print("\n" + "=" * 60)
    print("Verifying installations...")
    print("=" * 60)
    
    required_packages = [
        "streamlit",
        "pandas", 
        "numpy",
        "scikit-learn",
        "textblob",
        "matplotlib",
        "seaborn",
        "joblib"
    ]
    
    installed = []
    missing = []
    
    try:
        import pkg_resources
        installed_packages = {pkg.key for pkg in pkg_resources.working_set}
        
        for package in required_packages:
            if package.lower() in installed_packages:
                installed.append(package)
            else:
                missing.append(package)
    except:
        # Fallback method
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                              capture_output=True, text=True)
        installed_packages = result.stdout.lower()
        
        for package in required_packages:
            if package.lower() in installed_packages:
                installed.append(package)
            else:
                missing.append(package)
    
    print(f"\n✅ Installed packages ({len(installed)}):")
    for package in installed:
        print(f"   • {package}")
    
    if missing:
        print(f"\n❌ Missing packages ({len(missing)}):")
        for package in missing:
            print(f"   • {package}")
        return False
    else:
        print("\n🎉 All required packages are installed!")
        return True

def main():
    print("=" * 60)
    print("GENAI NEWS INTELLIGENCE SYSTEM - DEPENDENCY INSTALLER")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("\n⚠️  Please upgrade to Python 3.8 or higher")
        return
    
    print("\n" + "=" * 60)
    print("Installing dependencies...")
    print("=" * 60)
    
    # Try to read from requirements.txt first
    packages_to_install = []
    if os.path.exists("requirements.txt"):
        print("Found requirements.txt file, reading packages...")
        try:
            with open("requirements.txt", "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        packages_to_install.append(line)
            print(f"Found {len(packages_to_install)} packages in requirements.txt")
        except Exception as e:
            print(f"Error reading requirements.txt: {e}")
            packages_to_install = []
    
    # If no requirements.txt or empty, use default packages
    if not packages_to_install:
        print("Using default package list...")
        packages_to_install = [
            "streamlit>=1.28.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "textblob>=0.18.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "joblib>=1.3.0"
        ]
    
    # Install packages
    success_count = 0
    fail_count = 0
    
    for package in packages_to_install:
        if install_package(package):
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "=" * 60)
    print("INSTALLATION SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully installed: {success_count} packages")
    print(f"❌ Failed to install: {fail_count} packages")
    
    all_installed = check_installation()
    
    print("\n" + "=" * 60)
    if all_installed:
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. 📊 Train the model: python notebooks/model_training.py")
        print("2. 🚀 Run the app: streamlit run app.py")
        print("3. 🌐 Open browser: http://localhost:8501")
    else:
        print("⚠️  SETUP INCOMPLETE")
        print("\nSome packages failed to install.")
        print("Try installing manually:")
        print("pip install streamlit pandas scikit-learn textblob")
        print("\nOr check Python version compatibility.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()