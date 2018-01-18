value_justification = 8
name_justification = 4


class Point(list):
        """ Class used to represent points. The main purpose is to give a name attribute to a list, used for clarity purposes. """

        def __init__(self, name, contents):
                self.name = name
                super(Point, self).__init__()
                for item in contents:
                        self.append(item)

        def __str__(self):
                return self.name.ljust(name_justification) + '(' + ','.join(
                        str(component).rjust(value_justification) for component in self) + ')'

        def __repr__(self):
                return self.__str__()
