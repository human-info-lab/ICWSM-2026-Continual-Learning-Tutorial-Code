clean:
	rm -rf output/
	rm -rf trainer_output/

train:
	uv run -m src.train

eval: evaluate

evaluate:
	uv run -m src.evaluate