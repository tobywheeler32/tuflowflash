test:
	bin/pytest

beautiful:
	bin/isort fews_3di/
	bin/black fews_3di/