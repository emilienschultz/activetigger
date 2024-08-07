import { range } from 'lodash';
import { FC, useCallback, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

import { ProjectPageLayout } from './layout/ProjectPageLayout';

export const ProjectAnnotationPage: FC = () => {
  const { projectName, elementId } = useParams();
  const navigate = useNavigate();

  console.log(elementId);

  const navigateToNextElement = useCallback(async () => {
    // change url using the new elementId
    // const newElementId = await  apiCall()
    //navigate('/project/${projectName}/annotate/newid');
  }, [projectName, navigate]);

  useEffect(() => {
    if (elementId === undefined) {
      // fetch next elementId
      // change url using the new elementId
      // navigate("/project/${projectName}/annotate/newid");
      //navigateToNextElement();
    } else {
      //fetch element information (text and labels)
    }
  }, [elementId]);

  // we must get the project annotation payload /element
  if (!projectName) return null;
  return (
    <ProjectPageLayout projectName={projectName} currentAction="annotate">
      Annotation {projectName}
      <div className="container">
        <div className="row">
          <div className="col-6">
            <div>
              <label>Mode de sélection</label>
              <select>
                <option>deterministic</option>
              </select>
            </div>
            <div>
              <label>label</label>
              <select disabled>
                <option>not available</option>
              </select>
            </div>
          </div>
          <div className="col-6">
            <label>Scope</label>
            <select>
              <option>tagged</option>
            </select>
          </div>
        </div>
        <div className="row">
          <h1>Du texte à annoter</h1>
        </div>
        <div className="row">
          <h2>labels</h2>
          <div className="d-flex flex-wrap gap-2">
            {range(10).map((i) => (
              <button
                key={i}
                className="btn btn-primary grow-1"
                onClick={() => {
                  // add tag to element
                  // if pas d'erreur
                  navigateToNextElement();
                }}
              >
                blabla {i}
              </button>
            ))}
          </div>
        </div>
        <hr />
        <details>
          <summary>
            <h2>Label management</h2>
          </summary>
          Plein de bordel
        </details>
      </div>
    </ProjectPageLayout>
  );
};
