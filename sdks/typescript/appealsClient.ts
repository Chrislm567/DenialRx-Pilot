import fetch, { Response } from "node-fetch";

type DraftPayload = {
  draft_id: string;
  patient_first_name: string;
  patient_last_name: string;
  member_id: string;
  payer_name: string;
  provider_npi: string;
  denial_code: string;
  scenario: string;
  clinical_summary: string;
  attachments?: string[];
};

type SubmitPayload = DraftPayload & {
  submitted_by: string;
  callback_url?: string;
};

export class AppealsClient {
  constructor(private baseUrl = "http://localhost:8000", private apiKey?: string) {}

  private headers() {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  async draft(payload: DraftPayload) {
    return this.post("/appeals/draft", payload);
  }

  async submit(payload: SubmitPayload) {
    return this.post("/appeals/submit", payload);
  }

  async status(appealId: string) {
    return this.get(`/appeals/${appealId}`);
  }

  async audit(appealId: string) {
    return this.get(`/appeals/${appealId}/audit`);
  }

  private async post(path: string, payload: unknown) {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(payload),
    });
    return this.parseResponse(response);
  }

  private async get(path: string) {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: "GET",
      headers: this.headers(),
    });
    return this.parseResponse(response);
  }

  private async parseResponse(response: Response) {
    if (!response.ok) {
      const body = await response.text();
      throw new Error(`Appeals API error ${response.status}: ${body}`);
    }
    return response.json();
  }
}
