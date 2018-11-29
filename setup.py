from setuptools import setup, find_packages

setup(name='sparsely_connected_keras',
      version='1.0.0',
      description='Sparsely-connected layers for Keras',
      author='Amir Alavi',
      url='https://github.com/AmirAlavi/sparsely-connected-keras',
      license='GPLv3',
      packages=find_packages(),
      install_requires=[
          'keras>=2',
          'numpy',
      ],
      python_requires='>=3'
      )
