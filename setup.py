from setuptools import setup, find_packages

setup(name='sparsely_connected_keras',
      version='0.1.0',
      description='Sparsely-connected layers for Keras',
      author='Amir Alavi',
      packages=find_packages(),
      install_requires=[
          'keras>=2',
          'numpy',
      ],
      python_requires='>=3'
)
