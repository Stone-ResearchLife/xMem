from __future__ import annotations

import uuid
from typing import Any, AnyStr, Dict, Optional


class CircularDoublyLinkedList:
    def __init__(self, value: Any):
        """Create a bi-directional link node.

        Args:
            value (Any): The value of the node.
        """
        self._prev: Optional[CircularDoublyLinkedList] = self
        self._next: Optional[CircularDoublyLinkedList] = self
        self._value: Any = value
        self._id = uuid.uuid4().hex

    @property
    def prev(self) -> CircularDoublyLinkedList:
        return self._prev

    @property
    def next(self) -> CircularDoublyLinkedList:
        return self._next

    @property
    def value(self) -> Any:
        return self._value

    @property
    def id(self) -> str:
        return self._id

    def insert_after(self, node: CircularDoublyLinkedList):
        """Insert a node after the current node.

        Args:
            node (CircularDoublyLinkedList): The node to be inserted.

        Returns:
            None

        """
        node._prev = self
        node._next = self._next
        self._next._prev = node
        self._next = node

    def insert_before(self, node: CircularDoublyLinkedList):
        """Insert a node before the current node.

        Args:
            node (CircularDoublyLinkedList): The node to be inserted.

        Returns:
            None

        """
        node._prev = self._prev
        node._next = self
        self._prev._next = node
        self._prev = node

    def remove(self):
        """Remove the current node.

        Returns:
            None
        """
        self._prev._next = self._next
        self._next._prev = self._prev
        self._prev = self
        self._next = self

    def search(self, value: Any) -> Optional[CircularDoublyLinkedList]:
        """Search a node by value.

        Args:
            value (Any): The value to be searched.

        Returns:
            Optional[CircularDoublyLinkedList]: The node with the value.
        """
        node = self
        while node.value != value and node.next != self:
            node = node.next
        return node if node.value == value else None

    def __eq__(self, other: CircularDoublyLinkedList) -> bool:
        """Check if two nodes are equal."""
        return self.id == other.id

    def __str__(self):
        """Return the string representation of the node."""
        return str(self.value)

    def __repr__(self):
        """Return the string representation of the node."""
        return f"CircularDoublyLinkedList({self.value})"



class NonCircularDoublyLinkedNode:
    def __init__(self, value: Any):
        """Create a non-circular doubly linked list node.

        Args:
            value (Any): The value of the node.
        """
        self._prev: Optional[NonCircularDoublyLinkedNode] = None
        self._next: Optional[NonCircularDoublyLinkedNode] = None
        self._value: Any = value

    @property
    def prev(self) -> Optional[NonCircularDoublyLinkedNode]:
        return self._prev

    @property
    def next(self) -> Optional[NonCircularDoublyLinkedNode]:
        return self._next

    @property
    def value(self) -> Any:
        return self._value

    def insert_after(self, node: NonCircularDoublyLinkedNode):
        """Insert a node after the current node.

        Args:
            node (NonCircularDoublyLinkedNode): The node to be inserted.

        Returns:
            None
        """
        node._prev = self
        node._next = self._next
        if self._next:
            self._next._prev = node
        self._next = node


    def insert_before(self, node: NonCircularDoublyLinkedNode):
        """Insert a node before the current node.

        Args:
            node (NonCircularDoublyLinkedNode): The node to be inserted.

        Returns:
            None
        """
        node._next = self
        node._prev = self._prev
        if self._prev:
            self._prev._next = node
        self._prev = node

    def remove(self):
        """Remove the current node from the list.

        Returns:
            None
        """
        if self._prev:
            self._prev._next = self._next
        if self._next:
            self._next._prev = self._prev
        self._prev = None
        self._next = None

    def search(self, value: Any) -> Optional[NonCircularDoublyLinkedNode]:
        """Search a node by value starting from the current node.

        Args:
            value (Any): The value to be searched.

        Returns:
            Optional[NonCircularDoublyLinkedNode]: The node with the value if found, else None.
        """
        node = self
        while node:
            if node.value == value:
                return node
            node = node.next
        return None

    def __str__(self):
        """Return the string representation of the node."""
        return str(self.value)

    def __repr__(self):
        """Return the string representation of the node."""
        return f"DoublyLinkedNode({self.value})"