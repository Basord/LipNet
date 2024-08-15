from setuptools import setup

setup(name='lipnet',
    version='0.1.7',  # Increment the version number
    description='End-to-end sentence-level lipreading',
    url='http://github.com/rizkiarm/LipNet',
    author='Muhammad Rizki A.R.M',
    author_email='rizki@rizkiarm.com',
    license='MIT',
    packages=['lipnet'],
    zip_safe=False,
    install_requires=[
        'Keras',
        'editdistance',
        'h5py',
        'matplotlib',
        'numpy',
        'python-dateutil',
        'scipy',
        'Pillow',
        'tensorflow',  # or 'tensorflow-gpu' if you have a GPU
        'nltk',
        'scikit-video',
        'dlib'
    ])