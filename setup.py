from setuptools import setup, find_packages

setup(
    name="reward",
    version="0.0.4",
    description="Reinforcement Learning library",
    url="https://github.com/lgvaz/reward",
    author="lgvaz",
    author_email="lucasgouvaz@gmail.com",
    packages=[package for package in find_packages() if package.startswith("torchrl")],
    zip_safe=False,
)
