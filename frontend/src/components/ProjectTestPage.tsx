import { FC, useCallback, useEffect, useMemo, useState } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';

import { useNavigate, useParams } from 'react-router-dom';
import { useAddAnnotation, useGetElementById, useGetNextElementId } from '../core/api';
import { useAppContext } from '../core/context';
import { ElementOutModel } from '../types';
import { ProjectPageLayout } from './layout/ProjectPageLayout';
import { SelectCurrentScheme } from './SchemesManagement';

/**
 * Test component page
 */
export const ProjectTestPage: FC = () => {
  const { projectName, elementId } = useParams();
  //const { authenticatedUser } = useAuth();
  const {
    appContext: { currentScheme, currentProject: project, history },
    setAppContext,
  } = useAppContext();

  const availableLabels = useMemo(() => {
    return currentScheme && project ? project.schemes.available[currentScheme] || [] : [];
  }, [project, currentScheme]);

  const [element, setElement] = useState<ElementOutModel | null>(null);

  const navigate = useNavigate();

  // hooks to manage element
  const { getNextElementId } = useGetNextElementId(
    projectName || null,
    currentScheme || null,
    history,
  );
  const { getElementById } = useGetElementById(projectName || null, currentScheme || null);

  // hooks to manage annotation
  const { addAnnotation } = useAddAnnotation(projectName || null, currentScheme || null);

  // generic method to apply a chosen label to an element
  const applyLabel = useCallback(
    (label: string, elementId?: string) => {
      if (elementId) {
        setAppContext((prev) => ({ ...prev, history: [...prev.history, elementId] }));
        addAnnotation(elementId, label).then(() =>
          // redirect to next element by redirecting wihout any id
          // thus the getNextElementId query will be dont after the appcontext is reloaded
          navigate(`/projects/${projectName}/annotate/`),
        );
        // does not do nothing as we remount through navigate reFetchStatistics();
      }
    },
    [setAppContext, addAnnotation, navigate, projectName],
  );

  // Manage the actualisation of the element
  const selectionConfigTest = {
    mode: 'test',
    sample: 'untagged',
    displayPrediction: false,
    frameSelection: false,
    frame: [],
  };

  useEffect(() => {
    if (elementId === undefined) {
      getNextElementId(selectionConfigTest).then((nextElementId) => {
        if (nextElementId) getElementById(nextElementId).then(setElement);
      });
    } else {
      //fetch element information (text and labels)
      getElementById(elementId).then(setElement);
    }
  }, [elementId, getNextElementId, getElementById, navigate, selectionConfigTest, projectName]);

  return (
    <ProjectPageLayout projectName={projectName || null} currentAction="test">
      <div className="container-fluid">
        <div className="alert alert-danger m-5">This page is on construction</div>
        <span className="explanations">Annotate the test data set to test the model</span>
        <div className="row">
          <div className="col-6">
            <SelectCurrentScheme />
          </div>
          <div className="col-6">X/N of the test dataset annotated</div>
        </div>
        <Tabs id="panel" className="mb-3" defaultActiveKey="annotation">
          <Tab eventKey="annotation" title="1. Annotate">
            <div className="col-11 annotation-frame my-4">
              <span>{element?.text.slice(0, element?.limit as number)}</span>
              <span className="text-out-context">
                {element?.text.slice(element?.limit as number)}
              </span>
            </div>
            <div className="row">
              <div className="d-flex flex-wrap gap-2 justify-content-center">
                {
                  // display buttons for label
                  availableLabels.map((i, e) => (
                    <button
                      type="button"
                      key={i}
                      value={i}
                      className="btn btn-primary grow-1"
                      onClick={(e) => {
                        applyLabel(e.currentTarget.value, elementId);
                      }}
                    >
                      {i} <span className="badge text-bg-secondary">{e + 1}</span>
                    </button>
                  ))
                }
              </div>
            </div>
          </Tab>
          <Tab eventKey="compute" title="2. Compute"></Tab>
        </Tabs>
      </div>
    </ProjectPageLayout>
  );
};
