from setuptools import setup, find_packages

setup(name='sparsely_connected_keras',
      version='2.0.0',
      description='Sparsely-connected layers for Keras',
      author='Ramprasad',
      url='https://github.com/ichbinram/sparsely-connected-keras',
      license='GPLv3',
      packages=find_packages(),
      install_requires=[
          'tensorflow>=2',
          'numpy',
      ],
      python_requires='>=3'
      )
