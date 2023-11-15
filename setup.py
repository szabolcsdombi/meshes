from setuptools import Extension, setup

ext = Extension(
    name='meshes',
    sources=['meshes.cpp'],
)

setup(
    name='meshes',
    version='0.1.1',
    ext_modules=[ext],
)
