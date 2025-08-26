import * as fs from "fs";
import * as path from "path";
import { strict as assert } from "assert";

describe("alita extension", () => {
  it("includes snippet file", () => {
    const snippet = path.join(__dirname, "../snippets/alita.json");
    assert.ok(fs.existsSync(snippet), "snippet file missing");
  });
});
