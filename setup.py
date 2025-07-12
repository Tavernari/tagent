from setuptools import find_packages, setup

setup(
    name="tagent",  # Replace with your project name
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        # List your project's runtime dependencies here
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "mypy>=1.0.0",
            "isort>=5.0.0"
        ],
    },
    entry_points={
        "console_scripts": [
            "tagent=main:main"
        ],
    },
)
