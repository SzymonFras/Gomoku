import math
import numpy as np
import copy
from players_logic import get_network_output


class Node:
    def __init__(self, parent=None, prior_prob=1.0):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        self.status = 0

    def is_expanded(self):
        return len(self.children) > 0

    def expand(self, action_probs):
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = Node(parent=self, prior_prob=prob)

    def update(self, value):
        self.visit_count += 1
        self.value_sum += value

    def q_value(self):
        if self.status == 1:
            return 1.0
        if self.status == -1:
            return -1.0
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visit_count, parent_q=0.0, c_puct=1.5, fpu_reduction=0.20):
        if self.status == 1:
            return 1e6
        if self.status == -1:
            return -1e6

        if self.visit_count == 0:
            q = max(-1.0, -parent_q - fpu_reduction)
        else:
            q = self.q_value()

        u = c_puct * self.prior_prob * math.sqrt(max(1, parent_visit_count)) / (1 + self.visit_count)
        return q + u


class MCTS:
    def __init__(self, model, num_simulations=25, c_puct=1.5):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def get_action_probs(self, engine, temperature=1.0, add_noise=False, dirichlet_alpha=0.25, noise_eps=0.25):
        root = Node()
        policy, root_value = self._evaluate(engine)

        if add_noise and len(policy) > 1:
            actions = list(policy.keys())
            probs = np.array([policy[a] for a in actions], dtype=np.float32)
            noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
            noisy_probs = (1.0 - noise_eps) * probs + noise_eps * noise
            policy = {a: float(p) for a, p in zip(actions, noisy_probs)}

        root.expand(policy)
        root.update(-float(root_value))

        for _ in range(self.num_simulations):
            if any(c.status == 1 for c in root.children.values()):
                break
            if root.children and all(c.status == -1 for c in root.children.values()):
                break

            node = root
            sim_engine = copy.deepcopy(engine)
            search_path = [node]

            while node.is_expanded() and not sim_engine.is_game_over:
                action, node = self._select_child(node)
                r, c = action // sim_engine.board_size, action % sim_engine.board_size
                sim_engine.make_move(r, c)
                search_path.append(node)

            value = 0.0
            is_terminal_win = False
            if not sim_engine.is_game_over:
                policy, value = self._evaluate(sim_engine)
                node.expand(policy)
            else:
                if sim_engine.winner == 'Draw':
                    value = 0.0
                else:
                    value = -1.0
                    is_terminal_win = True

            self._backpropagate(search_path, float(value), is_terminal_win)

        action_probs = np.zeros(engine.board_size * engine.board_size, dtype=np.float32)

        winning_actions = [a for a, child in root.children.items() if child.status == 1]
        if winning_actions:
            action_probs[winning_actions[0]] = 1.0
            return action_probs

        for action, child in root.children.items():
            if child.status != -1:
                action_probs[action] = child.visit_count

        if np.sum(action_probs) == 0:
            for action, child in root.children.items():
                action_probs[action] = max(1, child.visit_count)

        max_visits = np.max(action_probs)
        if max_visits > 2:
            action_probs[action_probs <= 1] = 0.0

        if temperature == 0:
            best_action = int(np.argmax(action_probs))
            action_probs = np.zeros_like(action_probs)
            action_probs[best_action] = 1.0
        else:
            action_probs = action_probs ** (1.0 / temperature)
            action_probs /= np.sum(action_probs)

        return action_probs

    def _select_child(self, node):
        for action, child in node.children.items():
            if child.status == 1:
                return action, child

        best_score = -float('inf')
        best_action = -1
        best_child = None
        parent_q = node.q_value()

        non_losing = [c for c in node.children.values() if c.status != -1]
        ignore_losing = len(non_losing) > 0

        for action, child in node.children.items():
            if ignore_losing and child.status == -1:
                continue
            score = child.ucb_score(node.visit_count, parent_q=parent_q, c_puct=self.c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def _backpropagate(self, search_path, value, is_terminal_win=False):
        if is_terminal_win:
            search_path[-1].status = 1

        for node in reversed(search_path):
            value = -value
            node.update(value)
            if node.children and node.status == 0:
                if any(c.status == 1 for c in node.children.values()):
                    node.status = -1
                elif all(c.status == -1 for c in node.children.values()):
                    node.status = 1

    def _get_active_legal_moves(self, engine, radius=2):
        board_np = np.array(engine.board)
        occupied = np.argwhere(board_np != '.')
        if len(occupied) == 0:
            center = engine.board_size // 2
            return [(center, center)]

        mask = np.zeros((engine.board_size, engine.board_size), dtype=bool)
        for r, c in occupied:
            r_min, r_max = max(0, r - radius), min(engine.board_size, r + radius + 1)
            c_min, c_max = max(0, c - radius), min(engine.board_size, c + radius + 1)
            mask[r_min:r_max, c_min:c_max] = True

        mask &= (board_np == '.')
        active_moves = [tuple(x) for x in np.argwhere(mask)]
        return active_moves if active_moves else engine.get_legal_moves()

    def _evaluate(self, engine):
        device = next(self.model.parameters()).device
        logits, value, _ = get_network_output(self.model, engine.board, engine.current_player, device)
        logits_np = logits.cpu().numpy()
        legal_moves = self._get_active_legal_moves(engine, radius=2)
        policy = {}

        legal_indices = [r * engine.board_size + c for r, c in legal_moves]
        if legal_indices:
            legal_logits = logits_np[legal_indices]
            exp_logits = np.exp(legal_logits - np.max(legal_logits))
            probs = exp_logits / np.sum(exp_logits)

            for action_idx, prob in zip(legal_indices, probs):
                policy[action_idx] = prob

        return policy, value