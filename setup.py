from setuptools import find_packages, setup


setup(
    name="autoresearch-trade-bot",
    version="0.1.0",
    description="Agentic research kernel for crypto trading strategies.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["pyarrow>=18,<20"],
)
