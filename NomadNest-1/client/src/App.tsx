import { Route, Switch } from "wouter";
import Home from "@/pages/Home";
import WorkoutDetail from "@/pages/WorkoutDetail";
import MealPlan from "@/pages/MealPlan";
import NotFound from "@/pages/not-found";
import Header from "@/components/Header";
import Footer from "@/components/Footer";

function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <Switch>
        <Route path="/" component={Home} />
        <Route path="/workout/:id" component={WorkoutDetail} />
        <Route path="/meal-plan/:id" component={MealPlan} />
        <Route component={NotFound} />
      </Switch>
      <Footer />
    </div>
  );
}

export default App;
