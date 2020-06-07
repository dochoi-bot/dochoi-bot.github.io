# Jekyll 환경세팅

내 환경 windows 10 home의 wsl

ubuntu 18.04 LTS



```shell
$ sudo apt-get install -y ruby-full build-essential zlib1g-dev
```

```shell
$echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
$echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
$echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
$source ~/.bashrc
```

gem파일과 호환성이 안맞을땐 아래방법이 있다.

```shell
$ gem install bundler
$ bundle install
$ bundle exec jekyll serve
```

