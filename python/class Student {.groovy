class Student {
    String name;
    int age;

    // Constructor
    Student(String n, int a) {
        name = n;
        age = a;
    }

    // Method to display student details
    void display() {
        System.out.println("Name: " + name + ", Age: " + age);
    }

    public static void main(String[] args) {
        // Create an array of Student objects
        Student[] students = new Student[3];
        students[0] = new Student("Alice", 20);
        students[1] = new Student("Bob", 22);
        students[2] = new Student("Charlie", 21);

        // Iterate through the array and display each student's details
        for (int i = 0; i < students.length; i++) {
            students[i].display();
        }
    }
}