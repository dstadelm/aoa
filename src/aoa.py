#!/usr/bin/env python3
import logging
from pathlib import Path

import networkx as nx
from fast_sugiyama import from_edges
from matplotlib import image as mpimg
from matplotlib import pyplot as plt

from aoa.model.cpm import calculate_cpm
from aoa.model.network import Network, create_network
from aoa.model.project import load_yaml_project
from aoa.transform.coloring_strategy import ColoringStrategies
from aoa.transform.dot import create_dot
from aoa.transform.plantuml import PlantUml
from aoa.transform.to_networkx import to_networkx

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(logging.DEBUG)
    file = Path("tests/artefacts/more_tricky.yaml")
    project = load_yaml_project(file)
    network = create_network(project.get_activities())
    calculate_cpm(network)
    # plantuml = PlantUml(network)
    # print(plantuml.get_txt())

    nwx = to_networkx(network)
    # print(nwx.edges)
    # pos = from_edges(nwx.edges).to_dict()
    # print(pos)
    # nx.draw_networkx(nwx, pos=pos, with_labels=True, node_size=150)
    gvz = create_dot(nwx, ColoringStrategies.exponential)
    #
    _ = gvz.draw(file.with_suffix(".png"))
    _ = gvz.draw(file.with_suffix(".svg"))
    image = mpimg.imread(file.with_suffix(".png"))
    plt.title(str(file.with_suffix("")))
    plt.axis("off")
    plt.imshow(image)
    plt.show()
