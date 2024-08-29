import { FC, useState } from 'react';

import { useTableDisagreement } from '../core/api';

import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-quartz.css';
import { useAppContext } from '../core/context';

/*
 * Manage disagreement in annotations
 */

export const AnnotationDisagreementManagement: FC<{
  projectSlug: string;
}> = ({ projectSlug }) => {
  const {
    appContext: { currentScheme, currentProject: project },
  } = useAppContext();

  const availableLabels =
    currentScheme && project ? project.schemes.available[currentScheme] || [] : [];

  // get disagreement table
  const { tableDisagreement } = useTableDisagreement(projectSlug, currentScheme);

  // state elements to validate
  const [changes, setChanges] = useState<{ [key: string]: string }>({});

  return (
    <div className="container-fluid">
      <div className="row">
        <div className="col-3 m-3">
          <button
            className="btn btn-primary"
            onClick={() => {
              console.log(changes);
              setChanges({});
            }}
          >
            Validate changes
          </button>
        </div>
      </div>
      {tableDisagreement?.map((element, index) => (
        <div className="alert alert-warning" role="alert" key={index}>
          <div className="row d-flex align-items-center">
            <div className="row col-10">
              <span className="badge bg-light text-dark">{element.id}</span>
              <span>{element.text}</span>
            </div>
            <div className="row col-2">
              {element.annotations && (
                <table>
                  {Object.entries(element.annotations).map(([key, value], _) => (
                    <tr key={key}>
                      <td className="pe-2">{key}</td>
                      <td className="pe-2">
                        <span className="badge rounded-pill bg-warning text-dark">{value}</span>
                      </td>
                    </tr>
                  ))}
                </table>
              )}
            </div>
            <div className="container mt-4">
              <select
                className="form-select"
                onChange={(event) =>
                  setChanges({ ...changes, [element.id as string]: event.target.value })
                }
              >
                <option></option>
                {availableLabels.map((e) => (
                  <option key={e}>{e}</option>
                ))}
              </select>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};
