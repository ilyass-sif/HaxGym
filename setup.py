from setuptools import setup, find_packages

setup(
    name="haxgym",
    version="0.1.0",
    description="A high-performance JAX-based RL environment for Haxball.",
    author="HaxBot Research",
    author_email="example@example.com",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "numpy",
        "typing-extensions"
    ],
    python_requires=">=3.8",
)
