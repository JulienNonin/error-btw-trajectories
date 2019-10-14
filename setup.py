from setuptools import setup

setup(name='trajectories_error',
      version='0.1',
      description='estimate the error between two trajectories',
      url='https://github.com/JulienNonin/error-btw-trajectories',
      author='Julien NONIN & Eric KALA',
      license='WTFPL',
      packages=['trajectories_error'],
      install_requires=["numpy","matplotlib"],
      
      zip_safe=False)