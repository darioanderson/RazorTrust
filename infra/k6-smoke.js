import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  scenarios: {
    steady: { executor: "constant-vus", vus: 10, duration: "60s" },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<500"],
  },
};

export default function () {
  const response = http.get(`${__ENV.BASE_URL}/health/live`);
  check(response, { "liveness is healthy": (value) => value.status === 200 });
  sleep(0.2);
}
