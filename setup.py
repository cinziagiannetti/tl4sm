from distutils.core import setup



setup(
  name = 'tl4sm',         # How you named your package folder (MyLib)
  packages = ['tl4sm'],   # Chose the same as "name"
  version = '0.19',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'This library performs transfer learning for time series classification between different data time series using a ConvLSTM2D auto-encoder architecture. Two transfer learning types are provided as options - fine-tuning a given number of layers, or re-using the pre-trained model weights as initialisation step. The output returns the f_score, accuracy, and training time.',   # Give a short description about your library
  author = 'Aniekan Essien & Cinzia Giannetti',                   # Type in your name
  author_email = 'nakessien@outlook.com',      # Type in your E-Mail
  url = 'https://github.com/nakessien/tl4sm',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/nakessien/tl4sm/archive/0.19.tar.gz',    # I explain this later on
  keywords = ['Transfer Learning', 'Time Series Classification', 'ConvLSTM2D', 'Smart Manufacturing', 'Industry 4.0'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'keras',
          'tensorflow',
          'scikit-learn'          
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
