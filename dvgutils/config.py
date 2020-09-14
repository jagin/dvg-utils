import os
import re
import yaml
import logging


# Credits: https://medium.com/swlh/python-yaml-configuration-with-environment-variables-parsing-77930f4273ac
def load_config(config_file, overwrites=None, tag='!ENV'):
    """Load a yaml configuration file and resolve any environment variables.

    The environment variables must have !ENV before them and be in this format
    to be parsed: ${VAR_NAME}.
    E.g.::

    database:
        host: !ENV ${HOST}
        port: !ENV ${PORT}
    app:
        log_path: !ENV '/var/${LOG_PATH}'
        something_else: !ENV '${AWESOME_ENV_VAR}/var/${A_SECOND_AWESOME_VAR}'

    :param str config_file: the path to the yaml config file
    :param str overwrites: list of property values to overwrite
    :param str tag: the tag to look for
    :returns: the dict configuration
    :rtype: dict[str, T]
    """

    logger = logging.getLogger(__name__)
    logger.info(f"Load configuration from {config_file}")

    # pattern for global vars: look for ${word}
    pattern = re.compile(".*?\\${(\\w+)}.*?")
    loader = yaml.SafeLoader

    # the tag will be used to mark where to start searching for the pattern
    # e.g. somekey: !ENV somestring${MYENVVAR}blah blah blah
    loader.add_implicit_resolver(tag, pattern, None)

    def constructor_env_variables(loader, node):
        """Extracts the environment variable from the node's value.

        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :returns: the parsed string that contains the value of the environment
        variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        if match:
            full_value = value
            for g in match:
                full_value = full_value.replace(
                    f"${{{g}}}", os.environ.get(g, g)
                )
            return full_value
        return value

    loader.add_constructor(tag, constructor_env_variables)

    with open(config_file) as conf_data:
        config = yaml.load(conf_data, Loader=loader)

    if overwrites:
        overwrite_config(config, overwrites)

    return config


def overwrite_config(config, overwrites):
    for overwrite in overwrites:
        props, value = overwrite.split("=")
        props = props.split(".")
        value = yaml.safe_load(value)
        overwrite = props + [value]
        cur = config
        for path_item in overwrite[:-2]:
            try:
                cur = cur[path_item]
            except KeyError:
                cur[path_item] = {}
                cur = cur[path_item]

        cur[overwrite[-2]] = overwrite[-1]
