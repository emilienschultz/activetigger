import { FC, useCallback, useEffect, useState } from 'react';
import { Tab, Tabs } from 'react-bootstrap';
import { MdOutlineDeleteOutline } from 'react-icons/md';
import { useParams } from 'react-router-dom';
import Select from 'react-select';
import { useAppContext } from '../../../src/core/context';
import { DisplayTableTopics, Row } from '../../components/DisplayTableTopics';
import { useDeleteBertopic, useGetBertopicProjection, useGetBertopicTopics } from '../../core/api';
import { BertopicForm } from '../forms/BertopicForm';
import { ProjectPageLayout } from '../layout/ProjectPageLayout';
import { BertopicVizSigma } from '../ProjectionVizSigma/bertopicVizSigma';

export const BertopicPage: FC = () => {
  const { projectName } = useParams();
  const {
    appContext: { currentProject },
  } = useAppContext();
  const deleteBertopic = useDeleteBertopic(projectName || null);
  const availableBertopic = currentProject ? currentProject.bertopic.available : [];
  const [currentBertopic, setCurrentBertopic] = useState<string | null>(null);
  const { topics, parameters, reFetchTopics } = useGetBertopicTopics(
    projectName || null,
    currentBertopic,
  );
  const { projection, reFetchProjection } = useGetBertopicProjection(
    projectName || null,
    currentBertopic,
  );
  const currentTraining = currentProject ? Object.entries(currentProject.bertopic.training) : null;
  const availableModels = currentProject ? currentProject.bertopic.models : [];
  useEffect(() => {
    reFetchTopics();
    reFetchProjection();
  }, [currentBertopic, reFetchTopics, reFetchProjection]);
  const setSelectedId = useCallback((id?: string) => {
    console.log(id);
  }, []);
  console.log(projection);
  return (
    <ProjectPageLayout projectName={projectName} currentAction="explore">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12">
            <Tabs id="panel" className="mt-3">
              <Tab eventKey="existing" title="Existing Bertopic">
                <div className="explanations">
                  Compute a Bertopic on the train dataset to identify the main topics.
                </div>
                {currentTraining && currentTraining?.length > 0 && (
                  <div className="alert alert-info m-2">
                    Current computation
                    <ul>
                      {Object.values(currentTraining).map(([k, v]) => (
                        <li key={k}>
                          User {k} : {(v as unknown as { progress: string })?.progress}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                <h4 className="subsection">Existing Bertopic</h4>
                <div className="d-flex w-50">
                  <Select
                    className="flex-grow-1 "
                    options={Object.keys(availableBertopic).map((e) => ({ value: e, label: e }))}
                    onChange={(e) => {
                      if (e) setCurrentBertopic(e.value);
                    }}
                    value={{ value: currentBertopic, label: currentBertopic }}
                  />
                  <button
                    className="btn btn p-0"
                    onClick={() => {
                      deleteBertopic(currentBertopic);
                      setCurrentBertopic(null);
                    }}
                  >
                    <MdOutlineDeleteOutline size={30} />
                  </button>
                </div>
                <details>
                  <summary>Parameters</summary>
                  {parameters && JSON.stringify(parameters, null, 2)}
                </details>
                {topics && <DisplayTableTopics data={(topics as Row[]) || []} />}
                <div className="m-2">Test</div>
                {projection && (
                  <div className="row m-2" style={{ height: '500px' }}>
                    <BertopicVizSigma
                      className={`col-12 border p-0 h-100`}
                      data={
                        (projection as {
                          id: unknown[];
                          x: unknown[];
                          y: unknown[];
                          cluster: string[];
                        }) || null
                      }
                      setSelectedId={setSelectedId}
                    />
                  </div>
                )}
              </Tab>
              <Tab eventKey="new" title="New Bertopic">
                <div className="explanations">UMAP and HDBSCAN are being used</div>
                <BertopicForm projectSlug={projectName || null} availableModels={availableModels} />
              </Tab>
            </Tabs>
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
