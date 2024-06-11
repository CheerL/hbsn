try:
    import torchvision

    torchvision.disable_beta_transforms_warning()
except (ImportError, AttributeError):
    pass
