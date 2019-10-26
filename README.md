# Churn_interactive_demo
Retail churn prediction interactive demo

Installation
------------

This app can run on Python 3.7+ Install it with ``pip``:

To get the dependencies: ::

  pip install -r requirements.txt 

Usage
-----
The app is based on streamlit. So just run its server: ::

 streamlit run app.py


::

    usage: pigar [-h] [-v] [-u] [-s NAME [NAME ...]] [-c [PATH]] [-l LOG_LEVEL]
                 [-i DIR [DIR ...]] [-p SAVE_PATH] [-P PROJECT_PATH]
                 [-o COMPARISON_OPERATOR]

    Python requirements tool -- pigar, it will do only one thing at each time.
    Default action is generate requirements.txt in current directory.

    optional arguments:
      -h, --help          show this help message and exit
      -v, --version       show pigar version information and exit
      -u, --update        update database, use it when pigar failed you, exit when
                          action done
      -s NAME [NAME ...]  search package name by import name, use it if you do not
                          know import name come from which package, exit when
                          action done
      -c [PATH]           check requirements for the latest version. If file path
                          not given, search *requirements.txt in current
                          directory, if not found, generate file requirements.txt,
                          exit when action done
      -l LOG_LEVEL        show given level log messages, argument can be (ERROR,
                          WARNING, INFO), case-insensitive
      -i DIR [DIR ...]    given a list of directory to ignore, relative directory,
                          *used for* -c and default action
      -p SAVE_PATH        save requirements in given file path, *used for* default
                          action
      -P PROJECT_PATH     project path, which is directory, *used for* default
                          action
      -o COMPARISON_OPERATOR
                          The comparison operator for versions, alternatives:
                          [==, ~=, >=]

App
-----
![alt text](app.png)