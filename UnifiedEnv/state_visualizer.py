import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import os


class StateVisualizer:
    def __init__(self):
        self.G = nx.DiGraph()
        self.transitions: List[Tuple[Dict, Tuple[int, int], Dict]] = []
        self.action_names = {0: "adopt", 1: "match", 2: "mine", 3: "override"}

    def _state_to_str(self, state: Dict) -> str:
        """Convert state to string representation for node labels"""
        return f"Attacker: {state['len_attacker_forking']}\nHonest: {state['len_honest_forking']}"

    def _action_to_str(self, action: Tuple[int, int]) -> str:
        """Convert action tuple to string representation for edge labels"""
        action_type, action_arg = action
        return f"{self.action_names[action_type]}\n(arg={action_arg})"

    def add_transition(
        self, current_state: Dict, action: Tuple[int, int], next_state: Dict
    ) -> None:
        """Record a state transition"""
        self.transitions.append((current_state, action, next_state))

    def save_visualization(self, id: int, episode_steps: int) -> None:
        """Generate and save visualization of the episode's state transitions"""
        # Clear previous graph
        self.G.clear()

        # Add nodes and edges for each transition
        for current_state, action, next_state in self.transitions:
            current_node = self._state_to_str(current_state)
            next_node = self._state_to_str(next_state)

            # Add nodes if they don't exist
            if not self.G.has_node(current_node):
                self.G.add_node(current_node)
            if not self.G.has_node(next_node):
                self.G.add_node(next_node)

            # Add edge with action label
            self.G.add_edge(current_node, next_node, label=self._action_to_str(action))

        # Set up the plot with larger size
        plt.figure(figsize=(20, 10))

        # Use hierarchical layout for left-to-right progression
        pos = nx.spring_layout(self.G, k=2, iterations=50)

        # Draw nodes with custom style
        nx.draw_networkx_nodes(
            self.G, pos, node_color="lightblue", node_size=3000, node_shape="s"
        )  # square nodes

        # Draw node labels
        nx.draw_networkx_labels(self.G, pos, font_size=10)

        # Draw edges with arrows
        nx.draw_networkx_edges(
            self.G, pos, edge_color="gray", arrowsize=20, arrowstyle="->"
        )

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(self.G, "label")
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=8)

        # Add title
        plt.title(f"State Transitions - Episode {id} (Steps: {episode_steps})")

        # Save the plot
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig(
            f"visualizations/episode_{id}_{episode_steps}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        # Clear transitions for next episode
        self.transitions = []
