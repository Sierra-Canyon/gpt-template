# micrograd/viz.py  (T08)
# Draws an expression graph built from your Value objects. This is the drawing code from the Karpathy lecture
# (trace and draw_dot), typed here so engine.py holds only the Value class. Nothing to edit in this file.
#   draw_dot(root).render("micrograd/expr", format="svg", cleanup=True)   writes micrograd/expr.svg
#   print_graph(root)                                                     the same graph as indented text, if `dot` will not install
from graphviz import Digraph

# In: the output Value of an expression.  Out: every Value in the expression, and every (child, parent) edge.
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:                 # _prev holds the Values that produced v
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

# In: the output Value.  Out: a graphviz drawing, left to right, one box per Value and one small circle per operation.
def draw_dot(root):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))                          # a unique name for the box: the object's address
        grad = getattr(n, "grad", None)           # before T08 Step 5 a Value has no grad yet
        label = f"{{ {n.label} | data {n.data:.4f} }}" if grad is None else f"{{ {n.label} | data {n.data:.4f} | grad {grad:.4f} }}"
        dot.node(name=uid, label=label, shape="record")
        if n._op:                                 # an operation gets its own small node, feeding the box
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for child, parent in edges:
        dot.edge(str(id(child)), str(id(parent)) + parent._op)
    return dot

# In: a Value and the indent depth.  Out: nothing. Prints the graph from the output back to the leaves, indented.
def print_graph(v, depth=0):
    grad = getattr(v, "grad", None)
    line = "    " * depth + f"{v.label or '?'} = {v.data:.4f}"
    if grad is not None:
        line += f"   grad {grad:.4f}"
    if v._op:
        line += f"   ({v._op})"
    print(line)
    for child in v._prev:
        print_graph(child, depth + 1)
