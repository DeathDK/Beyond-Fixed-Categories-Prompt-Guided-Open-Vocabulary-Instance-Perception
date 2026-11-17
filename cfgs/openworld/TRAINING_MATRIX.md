# Combinatorial leave-out training matrix

Total known models (incl. F5): **31**

| Split | K | Known | Left-out (YOLOE prompts) | Config |
|---|---:|---|---|---|
| F5 | 5 | Car,Truck,Bus,Bicycle,Person | — | `f5.yaml` |
| K4-1 | 4 | Truck, Bus, Bicycle, Person | Car | `k4_1_leave_car.yaml` |
| K4-2 | 4 | Car, Bus, Bicycle, Person | Truck | `k4_2_leave_truck.yaml` |
| K4-3 | 4 | Car, Truck, Bicycle, Person | Bus | `k4_3_leave_bus.yaml` |
| K4-4 | 4 | Car, Truck, Bus, Person | Bicycle | `k4_4_leave_bicycle.yaml` |
| K4-5 | 4 | Car, Truck, Bus, Bicycle | Person | `k4_5_leave_person.yaml` |
| K3-1 | 3 | Bus, Bicycle, Person | Car, Truck | `k3_1_leave_car_truck.yaml` |
| K3-2 | 3 | Truck, Bicycle, Person | Car, Bus | `k3_2_leave_car_bus.yaml` |
| K3-3 | 3 | Truck, Bus, Person | Car, Bicycle | `k3_3_leave_car_bicycle.yaml` |
| K3-4 | 3 | Truck, Bus, Bicycle | Car, Person | `k3_4_leave_car_person.yaml` |
| K3-5 | 3 | Car, Bicycle, Person | Truck, Bus | `k3_5_leave_truck_bus.yaml` |
| K3-6 | 3 | Car, Bus, Person | Truck, Bicycle | `k3_6_leave_truck_bicycle.yaml` |
| K3-7 | 3 | Car, Bus, Bicycle | Truck, Person | `k3_7_leave_truck_person.yaml` |
| K3-8 | 3 | Car, Truck, Person | Bus, Bicycle | `k3_8_leave_bus_bicycle.yaml` |
| K3-9 | 3 | Car, Truck, Bicycle | Bus, Person | `k3_9_leave_bus_person.yaml` |
| K3-10 | 3 | Car, Truck, Bus | Bicycle, Person | `k3_10_leave_bicycle_person.yaml` |
| K2-1 | 2 | Bicycle, Person | Car, Truck, Bus | `k2_1_leave_car_truck_bus.yaml` |
| K2-2 | 2 | Bus, Person | Car, Truck, Bicycle | `k2_2_leave_car_truck_bicycle.yaml` |
| K2-3 | 2 | Bus, Bicycle | Car, Truck, Person | `k2_3_leave_car_truck_person.yaml` |
| K2-4 | 2 | Truck, Person | Car, Bus, Bicycle | `k2_4_leave_car_bus_bicycle.yaml` |
| K2-5 | 2 | Truck, Bicycle | Car, Bus, Person | `k2_5_leave_car_bus_person.yaml` |
| K2-6 | 2 | Truck, Bus | Car, Bicycle, Person | `k2_6_leave_car_bicycle_person.yaml` |
| K2-7 | 2 | Car, Person | Truck, Bus, Bicycle | `k2_7_leave_truck_bus_bicycle.yaml` |
| K2-8 | 2 | Car, Bicycle | Truck, Bus, Person | `k2_8_leave_truck_bus_person.yaml` |
| K2-9 | 2 | Car, Bus | Truck, Bicycle, Person | `k2_9_leave_truck_bicycle_person.yaml` |
| K2-10 | 2 | Car, Truck | Bus, Bicycle, Person | `k2_10_leave_bus_bicycle_person.yaml` |
| K1-1 | 1 | Person | Car, Truck, Bus, Bicycle | `k1_1_leave_car_truck_bus_bicycle.yaml` |
| K1-2 | 1 | Bicycle | Car, Truck, Bus, Person | `k1_2_leave_car_truck_bus_person.yaml` |
| K1-3 | 1 | Bus | Car, Truck, Bicycle, Person | `k1_3_leave_car_truck_bicycle_person.yaml` |
| K1-4 | 1 | Truck | Car, Bus, Bicycle, Person | `k1_4_leave_car_bus_bicycle_person.yaml` |
| K1-5 | 1 | Car | Truck, Bus, Bicycle, Person | `k1_5_leave_truck_bus_bicycle_person.yaml` |
