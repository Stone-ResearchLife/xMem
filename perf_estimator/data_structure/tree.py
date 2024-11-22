from __future__ import annotations
import uuid
from typing import Any, AnyStr, Dict, Iterator, List, Optional


class TreeNode:
    def __init__(self, value: Any):
        """Initialize a TreeNode object.

        Args:
            value (Any): The value of the node.
        """
        self._parent: Optional[TreeNode] = None
        self._children: Dict[AnyStr, TreeNode] = {}
        self._value: Any = value
        self._id = uuid.uuid4().hex

    @property
    def parent(self) -> Optional[TreeNode]:
        return self._parent

    @property
    def children(self) -> Dict[AnyStr, TreeNode]:
        return self._children

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def value(self) -> Any:
        return self._value

    @property
    def id(self) -> str:
        return self._id

    def add_child(self, child: TreeNode):
        """Add a child node to the current node.

        Args:
            child (TreeNode): The child node to be added.

        Returns:
            None
        """
        if child.id not in self.children.keys():
            self._children[child.id] = child
            child.set_parent(self)

    def remove_child(self, child: TreeNode):
        """Remove a child node from the current node.

        Args:
            child (TreeNode): The child node to be removed.

        Returns:
            None

        """
        if child.id in self.children.keys():
            self._children.pop(child.id)
            child.set_parent(None)

    def set_parent(self, parent: Optional[TreeNode]):
        """Set the parent node of the current node.

        Args:
            parent (Optional[TreeNode]): The parent node to be set.

        Returns:
            None
        """
        if parent is not None:
            if isinstance(self.parent, TreeNode):
                self.parent.remove_child(self)
        self._parent = parent

    def backward_stack(self) -> Iterator[TreeNode]:
        """Get the backward stack of the current node. The direction is from the current node to the root node."""
        current = self
        while current is not None:
            yield current
            current = current.parent

    def forward_stack(self, **kwargs) -> List[List[Any]]:
        """Get the forward stack of the current node. The direction is from the current node to the leaf node."""
        all_paths = []
        self._dfs(self, [], all_paths, **kwargs)
        return all_paths

    def _dfs(self, node: TreeNode, current_path: list, all_paths: list, **kwargs):
        assert isinstance(node, TreeNode)
        _attr = kwargs.get("attr", None)
        _key = node
        if _attr is not None:
            _key = getattr(node, _attr)
        current_path.append(_key)

        if not node.children:
            all_paths.append(list(current_path))
        else:
            for child in node.children.values():
                self._dfs(child, current_path, all_paths, **kwargs)
        # pop up the last index of the current path, and go back to the parent node
        # Example:
        # current_path = [1, 2, 3]
        # after pop: [1, 2]
        current_path.pop()
