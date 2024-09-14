from setuptools import setup, find_packages

setup(
    name='phondetection',  # パッケージ名
    version='0.1',         # バージョン
    packages=find_packages(),  # サブディレクトリも自動的に含める
    include_package_data=True,  # 静的ファイルやテンプレートも含める
    install_requires=[
        'Flask',           # Flask
        'opencv-python',   # OpenCV
        'numpy',           # NumPy
        'tensorflow',      # TensorFlow
    ],
    entry_points={
        'console_scripts': [
            'phondetection=phondetection.app:main',  # アプリの起動エントリーポイント
        ],
    },
    author='Your Name',
    author_email='youremail@example.com',
    description='A phone detection system using Flask, TensorFlow, and OpenCV.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/phondetection',  # GitHubのURL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
