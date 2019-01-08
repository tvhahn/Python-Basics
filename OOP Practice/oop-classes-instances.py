class Employee:
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + "." + last + "@company.com"

    def fullname(self):
        # Use self so it workw with all instances
        return "{} {}".format(self.first, self.last)


emp_1 = Employee("Corey", "Schafer", 50000)
emp_2 = Employee("Test", "User", 60000)

print(emp_1.email)
print(emp_2.email)

# Can use a print format to print employee names:
# print(
#     "{} {}".format(emp_1.first, emp_1.last)
# )  # Instead, create a method in our class to print

# Use instance to call method to print full name
print(emp_1.fullname())

# Call method on class, need to pass in the instance
print(Employee.fullname(emp_1))
