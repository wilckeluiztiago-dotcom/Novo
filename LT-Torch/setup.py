from setuptools import setup, find_packages

setup(
    name="lt_torch",
    version="0.1.0",
    description="Mini biblioteca de redes neurais estilo PyTorch em português",
    packages=find_packages(),
    install_requires=["numpy"],
)
