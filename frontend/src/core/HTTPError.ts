export class HttpError extends Error {
  status: Response['status'];

  constructor(status: Response['status'], message: string) {
    super(message);
    this.status = status;

    // Set the prototype explicitly.
    Object.setPrototypeOf(this, HttpError.prototype);
  }

  toString() {
    return `${this.status}: ${this.message}`;
  }
}
