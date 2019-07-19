from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name             = 'pce',
      version          = '0.1',
      author           = 'Anthony M. DeGennaro',
      author_email     = 'adegennaro@bnl.gov',
      description      = 'Python tools for various PCE tools',
      long_description = readme(),
      classifiers      = [
        'Topic :: Reduced Order Modeling :: PCE',
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: ISC License',
        'Programming Language :: Python :: 3.6',
      ],
      keywords         = 'PCE uncertainty quantification',
      url              = 'http://github.com/adegenna/pce',
      license          = 'ISC',
      packages         = ['pce','pce.src'],
      package_dir      = {'pce'         : 'pce' , \
                          'pce.src'     : 'pce/src' },
      #test_suite       = 'dmd.tests',
      #entry_points     = { 'console_scripts': ['Package = dmd.tests.testTwoState:main', 'PackageCH = dmd.tests.testCH:main'] },
      install_requires = [ 'numpy', 'scipy', 'matplotlib' ],
      python_requires  = '>=3',
      zip_safe         = False
)
