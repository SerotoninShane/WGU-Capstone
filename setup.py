from setuptools import setup, find_packages

try:
    long_description = open('README.md').read()
except FileNotFoundError:
    long_description = 'A book recommendation system based on KNN and cosine similarity.'

setup(
    name='recommended',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'scikit-learn',
        'pandas',
        'numpy',
    ],
    python_requires='>=3.7',
    description='A book recommendation system based on KNN and Cosine similarity.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Shane Bogue',
    author_email='sbogue8@wgu.edu',
    url='https://github.com/SerotoninShane/WGU-Capstone',
)
