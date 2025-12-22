class Rule:
    def __init__(self, index, context):
        self.index = index
        self.context = context

    def set_context(self, context):
        self.context = context

    def get_context(self):
        return self.context

    def set_index(self, index):
        self.index = index

    def get_index(self):
        return self.index

    def __lt__(self, other):
        # 当前规则的index更大，优先级更高
        return self.index > other.index

    def __str__(self):
        context_str = ', '.join(str(pred) for pred in self.context)
        return f"[index={self.index}, context={{ {context_str} }}]"

    def add_constraint(self, constraint):
        self.context.add(constraint)


class Predicate:
    def __init__(self, first, second, third):
        self.first = first
        self.second = second
        self.third = third

    def get_first(self):
        return self.first

    def get_second(self):
        return self.second

    def get_third(self):
        return self.third

    def __str__(self):
        return f"({self.first} {self.second} {self.third})"

    def __eq__(self, other):
        if isinstance(other, Predicate):
            return self.first == other.get_first() and self.second == other.get_second() and self.third == other.get_third()
        return False

    def __hash__(self):
        return hash(self.first) + hash(self.second) + hash(self.third)


def add_pred(self, rule, pred):
    temp = rule.get_context()
    res = set(temp)
    res.add(pred)
    return res


# predList = []
# p = Predicate("v4", "<", "30")
# predList.append(p)
# r = Rule(1, predList)
# r_list = {}
# r_list[0] = r
# for r1 in r_list.get(0):
#     print(r1.__str__())
