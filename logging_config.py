import logging
#from main import trace_id_var  # Import the context variable from your main file
import main

# 1. Create a custom filter to inject the traceId
class TraceIdFilter(logging.Filter):
    """
    A logging filter that injects the traceId from a context variable
    into the log record.
    """

    def filter(self, record):
        # Get the traceId from the context variable and add it to the log record
        record.traceId = main.trace_id_var.get()
        return True


# 2. Define the dictionary-based logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            # The magic happens here: we use the custom attribute '%(traceId)s'
            "format": "%(asctime)s [%(traceId)s] %(levelname)-8s [%(threadName)s] %(name)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        # Add a new formatter specifically for colored console output
        "colored": {
            # Use the special formatter class from colorlog
            "()": "colorlog.ColoredFormatter",
            # Add color codes to the format string
            "format": "%(asctime)s [%(traceId)s] %(log_color)s%(levelname)-8s%(reset)s [%(threadName)s] %(name)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "filters": {
        # Register our custom filter
        "trace_id_filter": {
            "()": TraceIdFilter,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "colored",
            "filters": ["trace_id_filter"],  # Attach the filter to the handler
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
    "loggers": {
        # Configure uvicorn loggers to use our format
        "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "hypercorn": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "hypercorn.access": {"handlers": ["console"], "level": "INFO", "propagate": False}
    }
}
