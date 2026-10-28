# LightGBM must be loaded before pyarrow, which pandas imports. Otherwise prediction
# crashes with an access violation on Windows. Importing it here covers every entry point.
import lightgbm  # noqa: F401
