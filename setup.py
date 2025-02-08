from setuptools import setup, find_packages

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
    description='A book recommendation system based on various algorithms like KNN and Cosine similarity.',
    long_description=open('README.txt').read(),
    long_description_content_type='text/plain',
    author='Shane Bogue',
    author_email='sbogue8@wgu.edu',
    url='https://github.com/SerotoninShane/WGU-Capstone',  # Replace with your GitHub link
)