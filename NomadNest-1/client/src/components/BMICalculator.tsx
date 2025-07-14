import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { 
  Card, 
  CardContent 
} from "@/components/ui/card";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Calculator } from "lucide-react";
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// BMI form schema
const bmiFormSchema = z.object({
  gender: z.enum(["male", "female"]),
  age: z.coerce.number().min(15, { message: "Age must be at least 15 years" }).max(100, { message: "Age must be 100 years or less" }),
  height: z.coerce.number().min(50, { message: "Height must be at least 50" }).max(300, { message: "Height must be 300 or less" }),
  heightUnit: z.enum(["cm", "ft"]),
  weight: z.coerce.number().min(20, { message: "Weight must be at least 20" }).max(500, { message: "Weight must be 500 or less" }),
  weightUnit: z.enum(["kg", "lb"]),
});

type BMIFormValues = z.infer<typeof bmiFormSchema>;

type BMIResult = {
  bmi: number;
  category: string;
  height: number;
  weight: number;
};

const BMICalculator = () => {
  const { toast } = useToast();
  const [bmiResult, setBmiResult] = useState<BMIResult | null>(null);
  const [isCalculating, setIsCalculating] = useState(false);

  const form = useForm<BMIFormValues>({
    resolver: zodResolver(bmiFormSchema),
    defaultValues: {
      gender: "male",
      age: undefined,
      height: undefined,
      heightUnit: "cm",
      weight: undefined,
      weightUnit: "kg",
    },
  });

  const onSubmit = async (data: BMIFormValues) => {
    setIsCalculating(true);
    try {
      const res = await apiRequest("POST", "/api/calculate-bmi", {
        height: data.height,
        weight: data.weight,
        heightUnit: data.heightUnit,
        weightUnit: data.weightUnit,
      });
      const result = await res.json();
      setBmiResult(result);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to calculate BMI. Please try again.",
        variant: "destructive",
      });
      console.error(error);
    } finally {
      setIsCalculating(false);
    }
  };

  // Calculate the percentage position for BMI marker
  const getBmiPercentage = (bmi: number): number => {
    if (bmi < 16) return 0;
    if (bmi > 40) return 100;
    
    // Maps BMI from 16-40 to 0-100%
    return ((bmi - 16) / (40 - 16)) * 100;
  };
  
  // Calculate circle progress for BMI visualization
  const getCircleProgress = (bmi: number): number => {
    // Calculate the circumference of the circle
    const circumference = 2 * Math.PI * 70;
    const percentage = getBmiPercentage(bmi);
    return circumference - (percentage / 100) * circumference;
  };

  // Get BMI description based on category
  const getBmiDescription = (category: string): string => {
    switch (category) {
      case "Underweight":
        return "You may need to gain some weight for better health.";
      case "Normal weight":
        return "Your BMI is within a healthy weight range.";
      case "Overweight":
        return "Losing some weight may benefit your health.";
      case "Obesity (Class 1)":
        return "Your health risks are increased. Consider weight loss.";
      case "Obesity (Class 2 & 3)":
        return "Your health risks are significantly increased.";
      default:
        return "";
    }
  };

  return (
    <section id="bmi-calculator" className="mb-12 scroll-mt-16">
      <div className="text-center mb-8">
        <h2 className="font-heading font-bold text-2xl md:text-3xl text-gray-800 mb-2">
          BMI Calculator
        </h2>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Measure your Body Mass Index to understand if your weight is healthy for your height.
        </p>
      </div>

      <Card className="max-w-4xl mx-auto">
        <CardContent className="p-0">
          <div className="md:flex">
            <div className="md:w-1/2 p-6 md:p-8">
              <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                  <FormField
                    control={form.control}
                    name="gender"
                    render={({ field }) => (
                      <FormItem className="space-y-2">
                        <FormLabel>Gender</FormLabel>
                        <FormControl>
                          <RadioGroup
                            onValueChange={field.onChange}
                            defaultValue={field.value}
                            className="flex space-x-4"
                          >
                            <div className="flex items-center space-x-2">
                              <RadioGroupItem value="male" id="male" />
                              <label htmlFor="male">Male</label>
                            </div>
                            <div className="flex items-center space-x-2">
                              <RadioGroupItem value="female" id="female" />
                              <label htmlFor="female">Female</label>
                            </div>
                          </RadioGroup>
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="age"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Age</FormLabel>
                        <FormControl>
                          <Input
                            type="number"
                            placeholder="Years"
                            min={15}
                            max={100}
                            {...field}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <div className="grid grid-cols-2 gap-4">
                    <FormField
                      control={form.control}
                      name="height"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Height</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              placeholder={form.watch("heightUnit") === "cm" ? "cm" : "feet"}
                              min={form.watch("heightUnit") === "cm" ? 50 : 1.5}
                              max={form.watch("heightUnit") === "cm" ? 300 : 9}
                              step={form.watch("heightUnit") === "cm" ? 1 : 0.01}
                              {...field}
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={form.control}
                      name="heightUnit"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Unit</FormLabel>
                          <Select
                            onValueChange={field.onChange}
                            defaultValue={field.value}
                          >
                            <FormControl>
                              <SelectTrigger>
                                <SelectValue placeholder="Select unit" />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              <SelectItem value="cm">Centimeters</SelectItem>
                              <SelectItem value="ft">Feet</SelectItem>
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <FormField
                      control={form.control}
                      name="weight"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Weight</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              placeholder={form.watch("weightUnit") === "kg" ? "kg" : "lb"}
                              min={form.watch("weightUnit") === "kg" ? 20 : 44}
                              max={form.watch("weightUnit") === "kg" ? 500 : 1100}
                              {...field}
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={form.control}
                      name="weightUnit"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Unit</FormLabel>
                          <Select
                            onValueChange={field.onChange}
                            defaultValue={field.value}
                          >
                            <FormControl>
                              <SelectTrigger>
                                <SelectValue placeholder="Select unit" />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              <SelectItem value="kg">Kilograms</SelectItem>
                              <SelectItem value="lb">Pounds</SelectItem>
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  </div>

                  <Button 
                    type="submit" 
                    className="w-full"
                    disabled={isCalculating}
                  >
                    {isCalculating ? "Calculating..." : "Calculate BMI"}
                  </Button>
                </form>
              </Form>
            </div>

            <div className="md:w-1/2 p-6 md:p-8 bg-gray-50">
              <div className="h-full flex flex-col items-center justify-center text-center">
                {!bmiResult ? (
                  <div className="text-gray-500">
                    <Calculator className="h-16 w-16 mb-4 text-primary opacity-30 mx-auto" />
                    <p>Enter your details and click calculate to see your BMI results.</p>
                  </div>
                ) : (
                  <div>
                    <div className="mb-4">
                      <div className="relative inline-block">
                        <svg className="w-40 h-40">
                          <circle 
                            cx="80" 
                            cy="80" 
                            r="70" 
                            fill="none" 
                            stroke="#e2e8f0" 
                            strokeWidth="12"
                          />
                          <circle 
                            cx="80" 
                            cy="80" 
                            r="70" 
                            fill="none" 
                            stroke="#3B82F6" 
                            strokeWidth="12" 
                            strokeDasharray="439.8 439.8" 
                            strokeDashoffset={getCircleProgress(bmiResult.bmi)} 
                            transform="rotate(-90, 80, 80)"
                            className="bmi-progress-circle"
                          />
                        </svg>
                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                          <span className="text-3xl font-bold">{bmiResult.bmi.toFixed(1)}</span>
                          <span className="text-sm text-gray-500">BMI</span>
                        </div>
                      </div>
                    </div>
                    <h3 className="text-xl font-semibold mb-1">{bmiResult.category}</h3>
                    <p className="text-gray-600 mb-4">{getBmiDescription(bmiResult.category)}</p>
                    <div className="w-full bmi-slider">
                      <div 
                        className="bmi-marker"
                        style={{ left: `${getBmiPercentage(bmiResult.bmi)}%` }}
                      ></div>
                    </div>
                    <div className="flex justify-between w-full text-xs mt-1 px-1">
                      <span>Underweight</span>
                      <span>Normal</span>
                      <span>Overweight</span>
                      <span>Obese</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </section>
  );
};

export default BMICalculator;
