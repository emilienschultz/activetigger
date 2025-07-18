import { FC, useMemo, useState } from 'react';
import Select from 'react-select';

import { useGetCompareSchemes } from '../core/api';
import { useAppContext } from '../core/context';

/*
 * Manage disagreement in annotations
 */

export const SchemesComparisonManagement: FC<{
  projectSlug: string;
}> = ({ projectSlug }) => {
  const {
    appContext: { currentScheme, currentProject: project },
  } = useAppContext();

  const [schemeA, setSchemeA] = useState<string | null>(currentScheme ? currentScheme : null);
  const [schemeB, setSchemeB] = useState<string | null>(null);
  const { compare } = useGetCompareSchemes(projectSlug, schemeA || '', schemeB || '');

  const availableSchemes = useMemo(() => {
    return project ? Object.keys(project.schemes.available) : [];
  }, [project]);

  return (
    <div className="container-fluid">
      <div className="row">
        <div className="col-12">
          <div className="explanations">Compare the annotations within 2 schemes</div>

          <div className="d-flex">
            <label className="form-label w-50">
              Scheme A
              <Select
                id="select-scheme-a"
                className="flex-grow-1"
                options={availableSchemes.map((e) => ({ value: e, label: e }))}
                value={{ value: schemeA, label: schemeA }}
                onChange={(selectedOption) => {
                  setSchemeA(selectedOption ? selectedOption.value : null);
                }}
                isClearable
                placeholder="Select scheme A"
              />
            </label>
            <label className="form-label w-50">
              Scheme B
              <Select
                id="select-scheme-b"
                className="flex-grow-1"
                options={availableSchemes.map((e) => ({ value: e, label: e }))}
                value={{ value: schemeB, label: schemeB }}
                onChange={(selectedOption) => {
                  setSchemeB(selectedOption ? selectedOption.value : null);
                }}
                isClearable
                placeholder="Select scheme B"
              />
            </label>
          </div>
          {/* <button className="btn btn-secondary w-25 m-1" onClick={reFetchCompare}>
            Compare
          </button> */}
        </div>
      </div>
      <div className="row">
        <div className="col-12">
          <div className="overflow-x-auto p-4">
            {compare ? (
              <table className="table-statistics">
                <head>
                  <tr>
                    <td>Metric</td>
                    <td>Value</td>
                  </tr>
                </head>
                <tbody>
                  <tr>
                    <td>Overlapping labels (%)</td>
                    <td>{compare.labels_overlapping}</td>
                  </tr>
                  <tr>
                    <td>Annotated elements in common</td>
                    <td>{compare.n_annotated}</td>
                  </tr>
                  <tr>
                    <td>Agreement Cohen-Kappa</td>
                    <td>{compare.cohen_kappa}</td>
                  </tr>
                  <tr>
                    <td>Agreement Percentage</td>
                    <td>{compare.percentage}</td>
                  </tr>
                </tbody>
              </table>
            ) : (
              <span>No data</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
