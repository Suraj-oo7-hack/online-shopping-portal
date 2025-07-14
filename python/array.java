class array
{
    String name;
    int age;
    Student(Stringn,inta)
    {
        name=n;
        age=a;

    }
    void display()
    {
        System.out.println("Name:" +name+ ", Age:" + age);
    }
     public static void main(String[]args)
{
    Student[] Students = new Student[3];
    Students[0] = new Student("Alice",20);
    Students[1] = new Student("Bob",22);
    Students[2] = new Student("charlie",21);
    for (int i=0;i<Students.length;i++)
    {
        Students[1].display();
    }
}
}
