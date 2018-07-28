from setuptools import setup, find_packages

setup(
    name="TorchRL",
    version="0.0.1",
    description="Reinforcement learning for OpenAI Gym and Pytorch",
    url="https://github.com/lgvaz/torchrl",
    author="lgvaz",
    author_email="lucasgouvaz@gmail.com",
    packages=[package for package in find_packages() if package.startswith("torchrl")],
    zip_safe=False,
)
