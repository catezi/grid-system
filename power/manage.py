#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import django
from django.conf import settings
# 设置环境变量
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "power.settings")
# 加载Django配置
django.setup()
# 设置Django项目路径，以便在其他文件中使用
settings.BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'power.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
